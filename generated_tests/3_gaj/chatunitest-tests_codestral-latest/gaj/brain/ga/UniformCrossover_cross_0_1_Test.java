package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class UniformCrossover_cross_0_1_Test {

    @Mock
    private Random rnd;

    @InjectMocks
    private UniformCrossover uniformCrossover;

    @Test
    void testCross() throws NoSuchFieldException, IllegalAccessException {
        // Mocking the Random object to return specific values
        when(rnd.nextBoolean()).thenReturn(true, false, true);
        // Creating mock genomes
        VectorGenome mom = mock(VectorGenome.class);
        VectorGenome dad = mock(VectorGenome.class);
        // Mocking the genes
        when(mom.getGenesCount()).thenReturn(3);
        when(mom.getGene(0)).thenReturn("gene1");
        when(mom.getGene(1)).thenReturn("gene2");
        when(mom.getGene(2)).thenReturn("gene3");
        when(dad.getGene(0)).thenReturn("gene4");
        when(dad.getGene(1)).thenReturn("gene5");
        when(dad.getGene(2)).thenReturn("gene6");
        // Mocking the evaluator
        Evaluator evaluator = mock(Evaluator.class);
        when(mom.getEvaluator()).thenReturn(evaluator);
        // Calling the method under test
        Genome result = uniformCrossover.cross(mom, dad);
        VectorGenome vSon = (VectorGenome) result;
        Field genesField = VectorGenome.class.getDeclaredField("genes");
        genesField.setAccessible(true);
        Vector<String> sonGenes = (Vector<String>) genesField.get(vSon);
        assertEquals(3, sonGenes.size());
        assertEquals("gene1", sonGenes.get(0));
        assertEquals("gene5", sonGenes.get(1));
        assertEquals("gene3", sonGenes.get(2));
        assertEquals(evaluator, vSon.getEvaluator());
    }
}
