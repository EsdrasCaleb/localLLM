package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class SectMutator_mutate_1_0_Test {

    @Mock
    private GAEnumAllelesSet allelesSet;

    @Mock
    private Random rnd;

    @InjectMocks
    private SectMutator mutator;

    @Test
    void mutate_withValidInput_updatesGenesProbabilistically() {
        // Set up mock for random numbers
        when(rnd.nextDouble()).thenReturn(0.1, 0.9, 0.2, 0.8, 0.0);
        // Create a VectorGenome instance (replace with a real one if possible)
        Vector<Integer> genes = new Vector<>();
        genes.add(1);
        genes.add(2);
        genes.add(3);
        // Use a real VectorGenome
        VectorGenome genome = new VectorGenome(genes);
        // Crucially, use the correct method to mock setting genes.
        // No need to mock getGenesCount
        // Test case 1: Mutation probability = 0.2
        int result = mutator.mutate(genome, 0.2);
        assertEquals(0, result);
        verify(genome, times(1)).setGene(0, anyInt());
        verify(genome, never()).setGene(1, anyInt());
        verify(genome, times(1)).setGene(2, anyInt());
        // Test case 2: Mutation probability = 0.8
        result = mutator.mutate(genome, 0.8);
        assertEquals(0, result);
        verify(genome, times(2)).setGene(anyInt(), anyInt());
        verify(genome, times(1)).setGene(0, anyInt());
        verify(genome, times(1)).setGene(1, anyInt());
        verify(genome, times(1)).setGene(2, anyInt());
        // Test case 3: Mutation probability = 0.0
        result = mutator.mutate(genome, 0.0);
        assertEquals(0, result);
        verify(genome, never()).setGene(anyInt(), anyInt());
    }

    @Test
    void mutate_withZeroProbability_doesNotMutateAnyGenes() {
        // Set up mocks and instance
        // Any value, it won't be used
        when(rnd.nextDouble()).thenReturn(0.5);
        Vector<Integer> genes = new Vector<>();
        genes.add(1);
        genes.add(2);
        genes.add(3);
        VectorGenome genome = new VectorGenome(genes);
        int result = mutator.mutate(genome, 0);
        assertEquals(0, result);
        verify(genome, never()).setGene(anyInt(), anyInt());
    }
}

// Dummy classes (replace with your actual classes)
class SectMutator {

    private GAEnumAllelesSet allelesSet;

    private Random rnd;

    public void setAllelesSet(GAEnumAllelesSet allelesSet) {
        this.allelesSet = allelesSet;
    }

    public int mutate(VectorGenome genome, double probability) {
        return 0;
    }

    public SectMutator() {
        this.rnd = new Random();
    }
}

class VectorGenome {

    private Vector<Integer> genes;

    public VectorGenome(Vector<Integer> genes) {
        this.genes = genes;
    }

    public int getGenesCount() {
        return genes.size();
    }

    public int getGene(int index) {
        return genes.get(index);
    }

    public boolean setGene(int index, int value) {
        if (index >= 0 && index < genes.size()) {
            genes.set(index, value);
            return true;
        }
        return false;
    }
}

class GAEnumAllelesSet {
}
