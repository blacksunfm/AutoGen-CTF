import json

a = """{
    "step_need_change": {
        "reason": "The current step is just starting and has not been successfully addressed yet.",
        "answer": false
    },
    "plan_need_change": {
        "reason": "The plan is currently being executed and there is no new information that suggests a change is necessary.",
        "answer": false
    },
    "next_speaker": {
        "reason": "The code_exec_agent is responsible for testing the identified vulnerabilities by crafting and sending specially crafted requests to the website.",
        "answer": "code\_exec\_agent"
    },
    "instruction_or_question": {
        "reason": "The code\_exec\_agent needs guidance on how to craft and send specially crafted requests to the website to test the identified vulnerabilities.",
        "answer": "Please craft and send specially crafted requests to the website to test the identified vulnerabilities. You can use tools such as curl or Burp Suite to send the requests. Make sure to follow the guidance provided by the review\_code\_agent in exploiting the vulnerabilities."
    }
}

"""
import json

b = json.loads(a)
print(b)
