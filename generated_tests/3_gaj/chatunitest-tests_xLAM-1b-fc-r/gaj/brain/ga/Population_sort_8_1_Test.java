package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_sort_8_1_Test {

    @Test
    public void sortTest() {
        // Given
        Population population = new Population();
        Genome genome1 = Mockito.mock(Genome.class);
        Genome genome2 = Mockito.mock(Genome.class);
        Genome genome3 = Mockito.mock(Genome.class);
        Genome genome4 = Mockito.mock(Genome.class);
        List<Genome> genomes = Arrays.asList(genome1, genome2, genome3, genome4);
        when(genome1.getScore()).thenReturn(1.0);
        when(genome2.getScore()).thenReturn(2.0);
        when(genome3.getScore()).thenReturn(3.0);
        when(genome4.getScore()).thenReturn(4.0);
        population.genoms = genomes;
        // When
        population.sort();
        // Then
        assertEquals(genome1, genomes.get(0));
        assertEquals(genome2, genomes.get(1));
        assertEquals(genome3, genomes.get(2));
        assertEquals(genome4, genomes.get(3));
    }
}
