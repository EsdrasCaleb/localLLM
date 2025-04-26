package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class Population_sort_8_4_Test {

    @Test
    public void testSort() {
        // Given
        Population population = new Population();
        Genome genome1 = mock(Genome.class);
        Genome genome2 = mock(Genome.class);
        when(genome1.getScore()).thenReturn(1.0);
        when(genome2.getScore()).thenReturn(2.0);
        List<Genome> genomes = new ArrayList<>();
        genomes.add(genome1);
        genomes.add(genome2);
        population.genoms = genomes;
        // When
        population.sort();
        // Then
        assertEquals(genome1, genomes.get(0));
        assertEquals(genome2, genomes.get(1));
    }
}
