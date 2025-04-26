package brain.ga;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class VectorGenome_getGene_2_0_Test {

    private VectorGenome genome;

    @BeforeEach
    void setUp() {
        genome = new VectorGenome();
        try {
            Field genesField = VectorGenome.class.getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(genome, new Vector<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Could not access genes field: " + e.getMessage());
        }
    }

    @Test
    void testGetGene_validIndex() {
        Vector<Object> genes = new Vector<>();
        genes.add("gene1");
        genes.add("gene2");
        genes.add("gene3");
        try {
            Field genesField = VectorGenome.class.getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(genome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Could not access genes field: " + e.getMessage());
        }
        assertEquals("gene1", genome.getGene(0));
        assertEquals("gene2", genome.getGene(1));
        assertEquals("gene3", genome.getGene(2));
    }

    @Test
    void testGetGene_indexOutOfBounds_throwsException() {
        Vector<Object> genes = new Vector<>();
        try {
            Field genesField = VectorGenome.class.getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(genome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Could not access genes field: " + e.getMessage());
        }
        assertThrows(IndexOutOfBoundsException.class, () -> genome.getGene(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> genome.getGene(1));
    }

    @Test
    void testGetGene_emptyVector() {
        Vector<Object> genes = new Vector<>();
        try {
            Field genesField = VectorGenome.class.getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(genome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Could not access genes field: " + e.getMessage());
        }
        assertThrows(IndexOutOfBoundsException.class, () -> genome.getGene(0));
    }
}
