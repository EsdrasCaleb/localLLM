package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import java.util.Vector;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // and returns one of them
java.util.*;

// The allele set class is a container for the different values that a gene may assume.
// If you call the allele member function with no argument,
// the allele set picks randomly from the alleles it contains
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
public class GAEnumAllelesSet_allele_0_1_Test {

    @Mock
    private Vector alleles;

    @InjectMocks
    private GAEnumAllelesSet enumAllelesSet;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testAllele() {
        when(alleles.size()).thenReturn(1);
        Object randomElement = new Object();
        when(alleles.get(anyInt())).thenReturn(randomElement);
        Object result = enumAllelesSet.allele();
        assertNotNull(result, "The allele method should not return null");
        assertEquals(randomElement, result, "The allele method should return a random element from the alleles vector");
    }
}
