package brain.ga;

import java.lang.reflect.Field;
import java.util.Random;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // and returns one of them
java.util.*;

// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains
@ExtendWith(MockitoExtension.class)
public class GAEnumAllelesSet_allele_0_0_Test {

    private GAEnumAllelesSet gaEnumAllelesSet;

    @Mock
    private Random mockRandom;

    private Vector<Object> alleles;

    @BeforeEach
    void setUp() {
        alleles = new Vector<>();
        gaEnumAllelesSet = new GAEnumAllelesSet();
        try {
            Field rndField = GAEnumAllelesSet.class.getDeclaredField("rnd");
            rndField.setAccessible(true);
            rndField.set(gaEnumAllelesSet, mockRandom);
            Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
            allelesField.setAccessible(true);
            allelesField.set(gaEnumAllelesSet, alleles);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set up mock Random and alleles: " + e.getMessage());
        }
    }

    @Test
    void testAlleleEmpty() {
        Object result = gaEnumAllelesSet.allele();
        assertNull(result);
    }

    @Test
    void testAlleleSingleElement() {
        alleles.add("test");
        when(mockRandom.nextInt(1)).thenReturn(0);
        Object result = gaEnumAllelesSet.allele();
        assertEquals("test", result);
    }

    @Test
    void testAlleleMultipleElements() {
        alleles.add("test1");
        alleles.add("test2");
        alleles.add("test3");
        when(mockRandom.nextInt(3)).thenReturn(1);
        Object result = gaEnumAllelesSet.allele();
        assertEquals("test2", result);
    }

    @Test
    void testAlleleWithNullElement() {
        alleles.add("test1");
        alleles.add(null);
        alleles.add("test3");
        when(mockRandom.nextInt(3)).thenReturn(1);
        Object result = gaEnumAllelesSet.allele();
        assertNull(result);
    }

    @Test
    void testAddAllele() {
        // Corrected:  There is no addAllele method.  We use setAlleles to manipulate the alleles.
        Vector<Object> newAlleles = new Vector<>();
        newAlleles.add("test");
        gaEnumAllelesSet.setAlleles(newAlleles);
        assertEquals(1, gaEnumAllelesSet.size());
        assertEquals("test", gaEnumAllelesSet.allele(0));
    }

    @Test
    void testAddAlleleNull() {
        // Corrected: There is no addAllele method. We use setAlleles to manipulate the alleles.
        Vector<Object> newAlleles = new Vector<>();
        newAlleles.add(null);
        gaEnumAllelesSet.setAlleles(newAlleles);
        assertEquals(1, gaEnumAllelesSet.size());
        assertNull(gaEnumAllelesSet.allele(0));
    }

    @Test
    @DisplayName("Test allele() with one allele")
    void testAlleleOne() {
        alleles.add("allele1");
        when(mockRandom.nextInt(1)).thenReturn(0);
        assertEquals("allele1", gaEnumAllelesSet.allele());
    }

    @Test
    @DisplayName("Test allele() with multiple alleles")
    void testAlleleMultiple() {
        alleles.add("allele1");
        alleles.add("allele2");
        alleles.add("allele3");
        when(mockRandom.nextInt(3)).thenReturn(1);
        assertEquals("allele2", gaEnumAllelesSet.allele());
    }
}
