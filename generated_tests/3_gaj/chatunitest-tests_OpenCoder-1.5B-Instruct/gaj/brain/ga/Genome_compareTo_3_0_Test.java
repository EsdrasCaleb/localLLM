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
        Genome g1 = Mockito.mock(Genome.class);
        Genome g2 = Mockito.mock(Genome.class);
        Mockito.when(g1.getScore()).thenReturn(3.0);
        Mockito.when(g2.getScore()).thenReturn(2.0);
        int result = g1.compareTo(g2);
        Assertions.assertEquals(-1, result);
    }
}
