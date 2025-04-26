package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class SectMutator_mutate_1_1_Test {

    @Test
    public void mutate_test(MockedStatic<SectMutator> mocked) {
        // given
        SectMutator sut = new SectMutator();
        Genome genome = mock(Genome.class);
        double pmut = 0.5;
        // when
        sut.mutate(genome, pmut);
        // then
        // no exception thrown
    }
}
