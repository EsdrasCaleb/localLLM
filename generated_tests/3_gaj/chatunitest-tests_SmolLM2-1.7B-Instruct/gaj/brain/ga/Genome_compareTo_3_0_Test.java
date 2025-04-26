package brain.ga;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@RunWith(MockitoJUnitRunner.class)
public class Genome_compareTo_3_0_Test {

    @Mock
    private Evaluator evaluator;

    @InjectMocks
    private Genome genome;

    @Test
    public void testCompareTo_HigherScore() {
        // Arrange
        when(genome.getScore()).thenReturn(10.0);
        when(genome.getEvaluator().evaluate(genome)).thenReturn(10.0);
        // Act
        int result = genome.compareTo(genome);
        // Assert
        assertEquals(1, result);
    }

    @Test
    public void testCompareTo_EqualScores() {
        // Arrange
        when(genome.getScore()).thenReturn(10.0);
        when(genome.getEvaluator().evaluate(genome)).thenReturn(10.0);
        // Act
        int result = genome.compareTo(genome);
        // Assert
        assertEquals(0, result);
    }

    @Test
    public void testCompareTo_LowerScore() {
        // Arrange
        when(genome.getScore()).thenReturn(5.0);
        when(genome.getEvaluator().evaluate(genome)).thenReturn(5.0);
        // Act
        int result = genome.compareTo(genome);
        // Assert
        assertEquals(-1, result);
    }
}
