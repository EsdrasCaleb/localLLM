package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Genome_compareTo_3_2_Test {

    @Test
    public void compareToTest() {
        // Given
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        genome1.evaluator = Mockito.mock(Evaluator.class);
        genome2.evaluator = Mockito.mock(Evaluator.class);
        Mockito.when(genome1.evaluator.evaluate(Mockito.any(Genome.class))).thenReturn(1.0);
        Mockito.when(genome2.evaluator.evaluate(Mockito.any(Genome.class))).thenReturn(2.0);
        // When
        int result = genome1.compareTo(genome2);
        // Then
        assertEquals(1, result);
    }
}
