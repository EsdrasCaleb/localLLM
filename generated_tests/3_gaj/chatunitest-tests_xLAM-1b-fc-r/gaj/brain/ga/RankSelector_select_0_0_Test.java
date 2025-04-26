package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class RankSelector_select_0_0_Test {

    @Test
    public void testSelect() {
        // Given
        RankSelector rankSelector = new RankSelector();
        Population population = Mockito.mock(Population.class);
        int populationSize = 10;
        int pos = 5;
        Genome genome = new Genome();
        Mockito.when(population.getSize()).thenReturn(populationSize);
        Mockito.when(population.get(pos)).thenReturn(genome);
        // When
        Genome selectedGenome = rankSelector.select(population);
        // Then
        assertEquals(genome, selectedGenome);
    }
}
