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

public class GAEnumAllelesSet_size_3_0_Test {

    @InjectMocks
    private GAEnumAllelesSet gaEnumAllelesSet;

    @Mock
    private Vector alleles;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testSize() {
        // Arrange
        when(alleles.size()).thenReturn(5);
        // Act
        int size = gaEnumAllelesSet.size();
        // Assert
        assertEquals(5, size);
    }

    @Test
    public void testSizeEmpty() {
        // Arrange
        when(alleles.size()).thenReturn(0);
        // Act
        int size = gaEnumAllelesSet.size();
        // Assert
        assertEquals(0, size);
    }
}
