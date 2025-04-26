package brain.ga;

import java.util.Vector;
import java.util.Random;
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

public class GAEnumAllelesSet_size_3_0_Test {

    @Test
    public void testSize() {
        GAEnumAllelesSet allelesSet = new GAEnumAllelesSet();
        Vector<Integer> alleles = new Vector<>();
        alleles.add(1);
        alleles.add(2);
        alleles.add(3);
        alleles.add(4);
        allelesSet.setAlleles(alleles);
        int expectedSize = 4;
        int actualSize = allelesSet.size();
        assertEquals(expectedSize, actualSize);
    }
}
