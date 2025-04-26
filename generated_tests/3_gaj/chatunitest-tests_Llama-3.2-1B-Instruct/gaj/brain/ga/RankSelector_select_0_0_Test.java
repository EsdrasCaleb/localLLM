package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class RankSelector_select_0_0_Test {

    @Mock
    private Population population;

    @InjectMocks
    private RankSelector focal;

    @Test
    public void testSelect() {
        // Given
        Random random = new Random();
        when(population.getSize()).thenReturn(10);
        // When
        Genome genome = focal.select(population);
        // Then
        assertNotNull(genome);
    }
}
