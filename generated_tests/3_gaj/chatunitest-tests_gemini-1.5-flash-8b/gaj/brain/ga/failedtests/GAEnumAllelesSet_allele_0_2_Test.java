package brain.ga;

import java.util.Random;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // and returns one of them
java.util.*;

// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains
class GAEnumAllelesSet_allele_0_2_Test {

    @Test
    void testAllele_emptyVector() {
        Vector<String> alleles = new Vector<>();
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        set.setAlleles(alleles);
        Object result = set.allele();
        assertEquals(null, result);
    }

    @Test
    void testAllele_singleElement() {
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        set.setAlleles(alleles);
        Object result = set.allele();
        assertEquals("A", result);
    }

    @Test
    void testAllele_multipleElements() {
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        set.setAlleles(alleles);
        Random mockRandom = Mockito.mock(Random.class);
        when(mockRandom.nextInt(alleles.size())).thenReturn(1);
        try {
            java.lang.reflect.Field rndField = GAEnumAllelesSet.class.getDeclaredField("rnd");
            rndField.setAccessible(true);
            rndField.set(set, mockRandom);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing or setting the 'rnd' field: " + e.getMessage());
        }
        Object result = set.allele();
        assertEquals("B", result);
    }

    @Test
    void testAllele_nonEmptyVector() {
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        set.setAlleles(alleles);
        Object result = set.allele();
        assertNotNull(result);
    }
}
