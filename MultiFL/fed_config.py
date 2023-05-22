#you can manually add more edge server and vehicles repectively by appending items
config = {
        'Edge0' :
        {
            'Agent0' :
            {
                'dataset' : '../datasets/3_Ren/',
            },

            'Agent1' :
            {
                'dataset' : '../datasets/5_Yang/',
            }
        },

        #'Edge1':
        #{
        #    'Agent0' :
        #    {
        #        'dataset' : '../datasets/CFD/',
        #    }#,

            #'Agent1' :
            #{
            #    'dataset' : '../datasets/CRACK500/',
            #},

            #'Agent2' :
            #{
            #    'dataset' : '../datasets/cracktree200/',
            #}
        #},

        'Edge1':
        {
            'Agent0' :
            {
                'dataset' : '../datasets/DeepCrack/',
            },

            'Agent1' :
            {
                'dataset' : '../datasets/Eugen/',
            },

            'Agent2' :
            {
                'dataset' : '../datasets/forest/',
            }
        },

        'Edge2':
        {
            'Agent0' :
            {
                'dataset' : '../datasets/Sylvie/',
            },

            'Agent1' :
            {
                'dataset' : '../datasets/Volker/',
            }#,

            #'Agent3' :
            #{
            #    'dataset' : '../datasets/GAPS384/',
            #}
            #'Agent0' :
            #{
            #    'dataset' : '../datasets/Rissbilder/',
            #},
        },

        'test':
        {
            'dataset' : '../datasets/',
        }
}
