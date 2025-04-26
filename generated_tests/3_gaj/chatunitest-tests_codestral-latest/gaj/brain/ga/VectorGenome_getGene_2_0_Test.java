package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class VectorGenome_getGene_2_0_Test {

    private VectorGenome vectorGenome;

    private Vector<Object> genes;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        vectorGenome = new VectorGenome();
        genes = new Vector<>();
        genes.add("gene1");
        genes.add("gene2");
        Field genesField = VectorGenome.class.getDeclaredField("genes");
        genesField.setAccessible(true);
        genesField.set(vectorGenome, genes);
    }

    @Test
    void testGetGene() {
        assertEquals("gene1", vectorGenome.getGene(0));
        assertEquals("gene2", vectorGenome.getGene(1));
    }

    @Test
    void testGetGeneOutOfBounds() {
        assertThrows(IndexOutOfBoundsException.class, () -> vectorGenome.getGene(2));
    }

    @Test
    void testGetGeneNegativeIndex() {
        assertThrows(IndexOutOfBoundsException.class, () -> vectorGenome.getGene(-1));
    }
}
