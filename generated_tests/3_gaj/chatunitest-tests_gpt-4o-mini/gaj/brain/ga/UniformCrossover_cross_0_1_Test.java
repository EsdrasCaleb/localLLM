package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class UniformCrossover_cross_0_1_Test {

    private UniformCrossover uniformCrossover;

    @Mock
    private Random mockRandom;

    @BeforeEach
    void setUp() {
        uniformCrossover = new UniformCrossover();
        try {
            Field rndField = UniformCrossover.class.getDeclaredField("rnd");
            rndField.setAccessible(true);
            rndField.set(uniformCrossover, mockRandom);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set up the test: " + e.getMessage());
        }
    }

    @Test
    void testCrossover() {
        // Mocking the random behavior
        when(mockRandom.nextBoolean()).thenReturn(true, false, true);
        VectorGenome parent1 = mock(VectorGenome.class);
        VectorGenome parent2 = mock(VectorGenome.class);
        when(parent1.getGenesCount()).thenReturn(3);
        when(parent1.getGene(0)).thenReturn(0.1);
        when(parent1.getGene(1)).thenReturn(0.2);
        when(parent1.getGene(2)).thenReturn(0.3);
        when(parent2.getGenesCount()).thenReturn(3);
        when(parent2.getGene(0)).thenReturn(0.4);
        when(parent2.getGene(1)).thenReturn(0.5);
        when(parent2.getGene(2)).thenReturn(0.6);
        Genome result = uniformCrossover.cross(parent1, parent2);
        VectorGenome vResult = (VectorGenome) result;
        assertEquals(3, vResult.getGenesCount());
        assertEquals(0.1, vResult.getGene(0));
        assertEquals(0.5, vResult.getGene(1));
        assertEquals(0.3, vResult.getGene(2));
    }

    @Test
    void testCrossoverWithDifferentRandom() {
        when(mockRandom.nextBoolean()).thenReturn(false, true, false);
        VectorGenome parent1 = mock(VectorGenome.class);
        VectorGenome parent2 = mock(VectorGenome.class);
        when(parent1.getGenesCount()).thenReturn(3);
        when(parent1.getGene(0)).thenReturn(0.1);
        when(parent1.getGene(1)).thenReturn(0.2);
        when(parent1.getGene(2)).thenReturn(0.3);
        when(parent2.getGenesCount()).thenReturn(3);
        when(parent2.getGene(0)).thenReturn(0.4);
        when(parent2.getGene(1)).thenReturn(0.5);
        when(parent2.getGene(2)).thenReturn(0.6);
        Genome result = uniformCrossover.cross(parent1, parent2);
        VectorGenome vResult = (VectorGenome) result;
        assertEquals(3, vResult.getGenesCount());
        assertEquals(0.4, vResult.getGene(0));
        assertEquals(0.2, vResult.getGene(1));
        assertEquals(0.6, vResult.getGene(2));
    }

    @Test
    void testCrossWithRandomSelectionFromMom() {
        VectorGenome mom = mock(VectorGenome.class);
        VectorGenome dad = mock(VectorGenome.class);
        when(mom.getGenesCount()).thenReturn(3);
        when(mom.getGene(0)).thenReturn("A");
        when(mom.getGene(1)).thenReturn("B");
        when(mom.getGene(2)).thenReturn("C");
        when(dad.getGene(0)).thenReturn("X");
        when(dad.getGene(1)).thenReturn("Y");
        when(dad.getGene(2)).thenReturn("Z");
        // Always select from mom
        when(mockRandom.nextBoolean()).thenReturn(true, true, true);
        Genome offspring = uniformCrossover.cross(mom, dad);
        assertTrue(offspring instanceof VectorGenome);
        VectorGenome vOffspring = (VectorGenome) offspring;
        assertEquals(3, vOffspring.getGenesCount());
        assertEquals("A", vOffspring.getGene(0));
        assertEquals("B", vOffspring.getGene(1));
        assertEquals("C", vOffspring.getGene(2));
    }

    @Test
    void testCrossWithRandomSelectionFromDad() {
        VectorGenome mom = mock(VectorGenome.class);
        VectorGenome dad = mock(VectorGenome.class);
        when(mom.getGenesCount()).thenReturn(3);
    }
}
