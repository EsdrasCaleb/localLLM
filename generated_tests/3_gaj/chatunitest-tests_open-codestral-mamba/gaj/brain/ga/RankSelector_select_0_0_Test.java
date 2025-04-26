package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class RankSelector_select_0_0_Test {

    @Mock
    private Population population;

    private RankSelector rankSelector;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
        rankSelector = new RankSelector();
    }

    @Test
    void select() {
        when(population.getSize()).thenReturn(10);
        when(population.get(anyInt())).thenReturn(mock(Genome.class));
        Genome result = rankSelector.select(population);
        assertNotNull(result);
    }
}
