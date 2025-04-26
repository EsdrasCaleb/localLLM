package brain.ga;

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

public class GAEnumAllelesSet_allele_1_1_Test {

    GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();

    @Test
    public void testAllele() {
        Vector<Object> alleles = new Vector<>();
        alleles.add("Allele1");
        alleles.add("Allele2");
        alleles.add("Allele3");
        alleles.add("Allele4");
        alleles.add("Allele5");
        gaEnumAllelesSet.setAlleles(alleles);
        assertEquals("Allele1", gaEnumAllelesSet.allele(0));
        assertEquals("Allele2", gaEnumAllelesSet.allele(1));
        assertEquals("Allele3", gaEnumAllelesSet.allele(2));
        assertEquals("Allele4", gaEnumAllelesSet.allele(3));
        assertEquals("Allele5", gaEnumAllelesSet.allele(4));
    }
}
