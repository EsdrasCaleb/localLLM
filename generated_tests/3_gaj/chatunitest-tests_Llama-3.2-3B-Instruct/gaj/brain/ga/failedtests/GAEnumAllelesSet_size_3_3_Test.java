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

public class GAEnumAllelesSet_size_3_3_Test {

    @Test
    public void testSize_EmptySet_ReturnsZero() {
        GAEnumAllelesSet ga = new GAEnumAllelesSet();
        int expected = 0;
        int actual = ga.size();
        assertEquals(expected, actual);
    }

    @Test
    public void testSize_NonEmptySet_ReturnsCorrectCount() {
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        GAEnumAllelesSet ga = new GAEnumAllelesSet();
        ga.setAlleles(alleles);
        int expected = 2;
        int actual = ga.size();
        assertEquals(expected, actual);
    }

    @Test
    public void testSize_NullAlleles_ThrowsNullPointerException() {
        GAEnumAllelesSet ga = new GAEnumAllelesSet();
        assertThrows(NullPointerException.class, () -> ga.size());
    }

    @Test
    public void testSize_NullGA_ThrowsNullPointerException() {
        assertThrows(NullPointerException.class, GAEnumAllelesSet::new);
    }
}
