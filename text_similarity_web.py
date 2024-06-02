from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    data = request.json
    job_description = data['job_description']
    profile_tag = data['profile_tag']
    
    job_embedding = model.encode([job_description], convert_to_tensor=True)
    profile_embedding = model.encode([profile_tag], convert_to_tensor=True)
    
    similarity_score = util.pytorch_cos_sim(job_embedding, profile_embedding)
    
    return jsonify({'similarity_score': similarity_score.item()})

if __name__ == '__main__':
    app.run(debug=True)