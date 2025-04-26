package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_4_Test {

    private VectorGenome vectorGenome;

    private Vector<Object> mockGenes;

    @BeforeEach
    public void setUp() {
        mockGenes = mock(Vector.class);
        vectorGenome = new VectorGenome(mockGenes, null);
    }

    @Test
    public void testGetGene() {
        int index = 5;
        Object expectedGene = "testGene";
        when(mockGenes.get(index)).thenReturn(expectedGene);
        Object actualGene = vectorGenome.getGene(index);
        assertEquals(expectedGene, actualGene);
    }
}
