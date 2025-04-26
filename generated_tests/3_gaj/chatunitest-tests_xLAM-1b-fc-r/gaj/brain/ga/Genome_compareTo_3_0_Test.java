package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Genome_compareTo_3_0_Test {

    @Test
    public void testCompareTo() {
        // Given
        Genome genome1 = new Genome();
        Genome genome2 = new Genome();
        genome1.evaluator = Mockito.mock(Evaluator.class);
        genome2.evaluator = Mockito.mock(Evaluator.class);
        Mockito.when(genome1.evaluator.evaluate(Mockito.any())).thenReturn(1.0);
        Mockito.when(genome2.evaluator.evaluate(Mockito.any())).thenReturn(2.0);
        // When
        int result = genome1.compareTo(genome2);
        // Then
        assertEquals(1, result);
        // Cleanup
        genome1 = null;
        genome2 = null;
    }
}
