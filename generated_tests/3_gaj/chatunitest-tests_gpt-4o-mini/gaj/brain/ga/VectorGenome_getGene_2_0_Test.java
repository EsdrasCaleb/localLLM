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

    @BeforeEach
    void setUp() {
        vectorGenome = new VectorGenome();
        // Using reflection to set the private 'genes' field
        try {
            Field genesField = VectorGenome.class.getDeclaredField("genes");
            genesField.setAccessible(true);
            Vector<Object> genes = new Vector<>();
            genes.add("Gene1");
            genes.add("Gene2");
            genes.add("Gene3");
            genesField.set(vectorGenome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set up genes field: " + e.getMessage());
        }
    }

    @Test
    void testGetGene_ValidIndex() {
        assertEquals("Gene1", vectorGenome.getGene(0));
        assertEquals("Gene2", vectorGenome.getGene(1));
        assertEquals("Gene3", vectorGenome.getGene(2));
    }

    @Test
    void testGetGene_InvalidIndex_Negative() {
        Exception exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            vectorGenome.getGene(-1);
        });
        assertEquals("Index: -1, Size: 3", exception.getMessage());
    }

    @Test
    void testGetGene_InvalidIndex_OutOfBounds() {
        Exception exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            vectorGenome.getGene(3);
        });
        assertEquals("Index: 3, Size: 3", exception.getMessage());
    }
}
