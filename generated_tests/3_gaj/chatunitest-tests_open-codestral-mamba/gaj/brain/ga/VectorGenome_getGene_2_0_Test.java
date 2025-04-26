package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class VectorGenome_getGene_2_0_Test {

    private VectorGenome vectorGenome;

    private Vector mockGenes;

    @BeforeEach
    void setUp() {
        mockGenes = mock(Vector.class);
        vectorGenome = new VectorGenome(mockGenes, null);
    }

    @Test
    void testGetGene() {
        Object expectedGene = new Object();
        when(mockGenes.get(1)).thenReturn(expectedGene);
        Object actualGene = vectorGenome.getGene(1);
        assertEquals(expectedGene, actualGene);
        verify(mockGenes, times(1)).get(1);
    }
}
