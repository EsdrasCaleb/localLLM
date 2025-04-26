package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_nextPos_1_0_Test {

    @Test
    public void testNextPos() throws Exception {
        // Create a mock object of GAUtilities
        GAUtilities gaUtilsMock = mock(GAUtilities.class);
        // Define the input values
        int n = 3;
        // Call the nextPos method on the mock object
        int result = gaUtilsMock.nextPos(n);
        // Verify the expected output
        assertEquals(2, result);
    }
}
