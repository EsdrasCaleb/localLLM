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

public class GAEnumAllelesSet_allele_0_0_Test {

    @Test
    public void testAllele() {
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector<Object> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        gaEnumAllelesSet.setAlleles(alleles);
        Random random = new Random();
        int randomIndex = random.nextInt(alleles.size());
        Object randomAllele = gaEnumAllelesSet.allele();
        assertEquals(alleles.get(randomIndex), randomAllele);
    }
}
