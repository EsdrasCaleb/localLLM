package brain.ga;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.lang.reflect.Field;
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

public class GAEnumAllelesSet_allele_1_0_Test {

    @Test
    void testAllele_validIndex() throws NoSuchFieldException, IllegalAccessException {
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        Vector<Object> alleles = new Vector<>();
        alleles.add("allele1");
        alleles.add("allele2");
        alleles.add("allele3");
        Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
        allelesField.setAccessible(true);
        allelesField.set(set, alleles);
        assertEquals("allele1", set.allele(0));
        assertEquals("allele2", set.allele(1));
        assertEquals("allele3", set.allele(2));
    }

    @ParameterizedTest
    // Test for invalid indices
    @ValueSource(ints = { -1, 3 })
    void testAllele_invalidIndex(int index) throws NoSuchFieldException, IllegalAccessException {
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        Vector<Object> alleles = new Vector<>();
        alleles.add("allele1");
        alleles.add("allele2");
        alleles.add("allele3");
        Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
        allelesField.setAccessible(true);
        allelesField.set(set, alleles);
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> set.allele(index));
    }

    @Test
    void testAllele_emptyAlleles() throws NoSuchFieldException, IllegalAccessException {
        GAEnumAllelesSet set = new GAEnumAllelesSet();
        Field allelesField = GAEnumAllelesSet.class.getDeclaredField("alleles");
        allelesField.setAccessible(true);
        allelesField.set(set, new Vector<>());
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> set.allele(0));
    }
}
