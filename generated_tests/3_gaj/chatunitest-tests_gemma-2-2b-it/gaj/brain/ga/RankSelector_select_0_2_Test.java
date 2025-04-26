package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class RankSelector_select_0_2_Test {

    @Test
    void select_shouldReturnRandomGenome() {
        RankSelector rankSelector = new RankSelector();
        Population population = mock(Population.class);
        when(population.getSize()).thenReturn(10);
        Genome expectedGenome = new Genome();
        when(population.get(anyInt())).thenReturn(expectedGenome);
        Genome actualGenome = rankSelector.select(population);
        assertEquals(expectedGenome, actualGenome);
    }
}
