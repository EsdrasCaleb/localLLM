package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_flipCoin_0_0_Test {

    @Test
    public void testFlipCoin() {
        // Arrange
        double prob = 0.5;
        // Act
        boolean result = GAUtilities.flipCoin(prob);
        // Assert
        assertEquals(true, result);
    }
}
