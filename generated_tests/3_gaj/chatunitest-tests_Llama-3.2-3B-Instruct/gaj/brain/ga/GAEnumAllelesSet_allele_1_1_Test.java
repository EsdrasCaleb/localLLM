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
public class GAEnumAllelesSet_allele_1_1_Test {

    @Test
    public void testAlleleReturnsAlleleOfExpectedType() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        alleles.add("Allele1");
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object allele = gaEnumAllelesSet.allele(0);
        // Assert
        assertEquals(String.class, allele.getClass());
    }

    @Test
    public void testAlleleReturnsAlleleOfExpectedTypeForInteger() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        alleles.add(1);
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object allele = gaEnumAllelesSet.allele(0);
        // Assert
        assertEquals(Integer.class, allele.getClass());
    }

    @Test
    public void testAlleleReturnsAlleleOfExpectedTypeForDouble() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        alleles.add(1.0);
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object allele = gaEnumAllelesSet.allele(0);
        // Assert
        assertEquals(Double.class, allele.getClass());
    }

    @Test
    public void testAlleleReturnsAlleleOfExpectedTypeForBoolean() {
        // Arrange
        GAEnumAllelesSet gaEnumAllelesSet = new GAEnumAllelesSet();
        Vector alleles = new Vector();
        alleles.add(true);
        gaEnumAllelesSet.setAlleles(alleles);
        // Act
        Object allele = gaEnumAllelesSet.allele(0);
        // Assert
        assertEquals(Boolean.class, allele.getClass());
    }
}
