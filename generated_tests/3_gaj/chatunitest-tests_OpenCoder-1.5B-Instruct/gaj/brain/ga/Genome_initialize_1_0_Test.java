package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Genome_initialize_1_0_Test {

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Genome genome;

    @Test
    public void testInitialize() throws Exception {
        // Arrange
        double score = 0.5;
        when(evaluator.evaluate(genome)).thenReturn(score);
        // This will initialize the evaluator
        genome.getEvaluator();
        // Act
        genome.initialize();
        // Assert
        assert genome.getScore() == score;
    }
}
