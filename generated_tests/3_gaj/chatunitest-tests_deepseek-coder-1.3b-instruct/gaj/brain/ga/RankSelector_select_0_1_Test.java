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
    private Population mockPopulation;

    @InjectMocks
    private RankSelector rankSelector;

    @Test
    public void selectTest() {
        // Given
        when(mockPopulation.get(anyInt())).thenReturn(mock(Genome.class));
        when(mockPopulation.getSize()).thenReturn(10);
        // When
        Genome genome = rankSelector.select(mockPopulation);
        // Then
        verify(mockPopulation, times(1)).get(anyInt());
        verify(mockPopulation, times(1)).getSize();
        assertNotNull(genome);
    }
}
