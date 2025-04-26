package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Genome_compareTo_3_1_Test {

    private Genome genome1;

    private Genome genome2;

    @BeforeEach
    void setUp() {
        genome1 = Mockito.mock(Genome.class);
        genome2 = Mockito.mock(Genome.class);
    }

    @Test
    void compareToTest() {
        Mockito.when(genome1.getScore()).thenReturn(0.8);
        Mockito.when(genome2.getScore()).thenReturn(0.6);
        assertEquals(1, genome1.compareTo(genome2));
        Mockito.when(genome1.getScore()).thenReturn(0.6);
        Mockito.when(genome2.getScore()).thenReturn(0.6);
        assertEquals(0, genome1.compareTo(genome2));
        Mockito.when(genome1.getScore()).thenReturn(0.6);
        Mockito.when(genome2.getScore()).thenReturn(0.8);
        assertEquals(-1, genome1.compareTo(genome2));
    }
}
