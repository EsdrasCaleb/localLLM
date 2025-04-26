package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class RankSelector_select_0_1_Test {

    @Mock
    private Population population;

    @Mock
    private Genome genome;

    @InjectMocks
    private RankSelector rankSelector;

    @BeforeEach
    public void setUp() {
        when(population.getSize()).thenReturn(5);
        when(population.get(anyInt())).thenReturn(genome);
    }

    @Test
    public void testSelect() {
        Genome selectedGenome = rankSelector.select(population);
        assertNotNull(selectedGenome);
        assertEquals(genome, selectedGenome);
    }
}
