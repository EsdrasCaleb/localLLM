package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_0_Test {

    @Test
    void testGetGene() {
        VectorGenome vectorGenome = new VectorGenome();
        vectorGenome.setGene(0, "Gene1");
        vectorGenome.setGene(1, "Gene2");
        assertEquals("Gene1", vectorGenome.getGene(0));
        assertEquals("Gene2", vectorGenome.getGene(1));
    }

    @Test
    void testGetGeneWithNegativeIndex() {
        VectorGenome vectorGenome = new VectorGenome();
        vectorGenome.setGene(-1, "Gene1");
        assertEquals("Gene1", vectorGenome.getGene(-1));
    }

    @Test
    void testGetGeneOutOfBounds() {
        VectorGenome vectorGenome = new VectorGenome();
        assertThrows(IndexOutOfBoundsException.class, () -> vectorGenome.getGene(Integer.MAX_VALUE));
    }
}
