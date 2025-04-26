package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class SectMutator_mutate_1_0_Test {

    @Mock
    private Random rnd;

    @Mock
    private GAEnumAllelesSet allelesSet;

    @Mock
    private VectorGenome genome;

    @InjectMocks
    private SectMutator sectMutator;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        when(genome.getGenesCount()).thenReturn(3);
        when(allelesSet.allele()).thenReturn("mockAllele");
    }

    @Test
    public void testMutateWithNoMutation() throws Exception {
        when(rnd.nextDouble()).thenReturn(0.1, 0.2, 0.3);
        double pmut = 0.05;
        int result = sectMutator.mutate(genome, pmut);
        assertEquals(0, result);
        verify(genome, never()).setGene(anyInt(), anyString());
    }

    @Test
    public void testMutateWithAllGenesMutated() throws Exception {
        when(rnd.nextDouble()).thenReturn(0.6, 0.6, 0.6);
        double pmut = 0.5;
        int result = sectMutator.mutate(genome, pmut);
        assertEquals(3, result);
        verify(genome, times(3)).setGene(anyInt(), anyString());
    }

    @Test
    public void testMutateWithSomeGenesMutated() throws Exception {
        when(rnd.nextDouble()).thenReturn(0.1, 0.6, 0.3);
        double pmut = 0.5;
        int result = sectMutator.mutate(genome, pmut);
        assertEquals(1, result);
        verify(genome, times(1)).setGene(anyInt(), anyString());
    }

    @Test
    public void testMutateWithPmutZero() throws Exception {
        when(rnd.nextDouble()).thenReturn(0.1, 0.2, 0.3);
        double pmut = 0.0;
        int result = sectMutator.mutate(genome, pmut);
        assertEquals(0, result);
        verify(genome, never()).setGene(anyInt(), anyString());
    }

    @Test
    public void testMutateWithPmutOne() throws Exception {
        when(rnd.nextDouble()).thenReturn(0.1, 0.2, 0.3);
        double pmut = 1.0;
        int result = sectMutator.mutate(genome, pmut);
        assertEquals(3, result);
        verify(genome, times(3)).setGene(anyInt(), anyString());
    }
}
