package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_1_Test {

    private VectorGenome vectorGenome;

    private Vector<Object> mockGenes;

    @BeforeEach
    public void setup() {
        mockGenes = mock(Vector.class);
        vectorGenome = new VectorGenome(mockGenes, null);
    }

    @Test
    public void testGetGene() {
        int index = 0;
        Object gene = "testGene";
        when(mockGenes.get(index)).thenReturn(gene);
        Object result = vectorGenome.getGene(index);
        assertEquals(gene, result);
    }
}
