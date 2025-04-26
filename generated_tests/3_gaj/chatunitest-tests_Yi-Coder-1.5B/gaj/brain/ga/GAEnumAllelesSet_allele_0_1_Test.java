package brain.ga;

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
@ExtendWith(MockitoExtension.class)
public class GAEnumAllelesSet_allele_0_1_Test {

    // Test class
    @Test
    public void testAllele() {
        // Create a new GAEnumAllelesSet object
        GAEnumAllelesSet ga = new GAEnumAllelesSet();
        // Set the alleles
        Vector alleles = new Vector();
        alleles.add("A");
        alleles.add("C");
        alleles.add("T");
        alleles.add("G");
        ga.setAlleles(alleles);
        // Get the allele
        Object allele = ga.allele();
        // Check if the allele is in the alleles vector
        assertTrue(alleles.contains(allele));
    }
}
