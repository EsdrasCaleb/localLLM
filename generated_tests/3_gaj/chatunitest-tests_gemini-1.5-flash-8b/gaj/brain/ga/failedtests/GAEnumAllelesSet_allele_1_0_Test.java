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

public class GAEnumAllelesSet_allele_1_0_Test {

    @Test
    public void testAllele_validIndex() {
        // Arrange
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        alleles.add("C");
        GAEnumAllelesSet sut = new GAEnumAllelesSet();
        sut.setAlleles(alleles);
        // Act
        Object result = sut.allele(1);
        // Assert
        assertEquals("B", result);
    }

    @Test
    public void testAllele_invalidIndex() {
        // Arrange
        Vector<String> alleles = new Vector<>();
        alleles.add("A");
        alleles.add("B");
        GAEnumAllelesSet sut = new GAEnumAllelesSet();
        sut.setAlleles(alleles);
        // Act
        Object result = sut.allele(2);
        // Assert
        assertNull(result);
    }

    @Test
    public void testAllele_emptyVector() {
        // Arrange
        Vector<String> alleles = new Vector<>();
        GAEnumAllelesSet sut = new GAEnumAllelesSet();
        sut.setAlleles(alleles);
        // Act
        Object result = sut.allele(0);
        // Assert
        assertNull(result);
    }

    @Test
    public void testAllele_nullVector() {
        // Arrange
        GAEnumAllelesSet sut = new GAEnumAllelesSet();
        sut.setAlleles(null);
        // Act
        Object result = sut.allele(0);
        // Assert
        assertNull(result);
    }
}
