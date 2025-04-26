package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class VectorGenome_getGene_2_0_Test {

    @Test
    void testGetGeneValidIndex() {
        // Arrange
        Vector<Object> genes = new Vector<>();
        genes.add("gene1");
        genes.add("gene2");
        VectorGenome vectorGenome = new VectorGenome();
        try {
            java.lang.reflect.Field genesField = vectorGenome.getClass().getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(vectorGenome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to access private field.");
        }
        int index = 1;
        // Act
        Object result = vectorGenome.getGene(index);
        // Assert
        assertEquals("gene2", result);
    }

    @Test
    void testGetGeneInvalidIndex() {
        // Arrange
        Vector<Object> genes = new Vector<>();
        genes.add("gene1");
        genes.add("gene2");
        VectorGenome vectorGenome = new VectorGenome();
        try {
            java.lang.reflect.Field genesField = vectorGenome.getClass().getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(vectorGenome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to access private field.");
        }
        // Invalid index
        int index = 10;
        // Act
        Object result = vectorGenome.getGene(index);
        // Assert
        assertNull(result, "Should return null for invalid index");
    }

    @Test
    void testGetGeneEmptyVector() {
        // Arrange
        Vector<Object> genes = new Vector<>();
        VectorGenome vectorGenome = new VectorGenome();
        try {
            java.lang.reflect.Field genesField = vectorGenome.getClass().getDeclaredField("genes");
            genesField.setAccessible(true);
            genesField.set(vectorGenome, genes);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to access private field.");
        }
        int index = 0;
        // Act
        Object result = vectorGenome.getGene(index);
        // Assert
        assertNull(result, "Should return null for empty vector");
    }
}
