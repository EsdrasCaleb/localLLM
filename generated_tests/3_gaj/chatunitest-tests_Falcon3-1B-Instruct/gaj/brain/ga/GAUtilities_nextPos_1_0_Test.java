package brain.ga;

import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_nextPos_1_0_Test {

    @Test
    public void testNextPos() {
        GAUtilities utilities = new GAUtilities();
        // Example expected position
        int expectedPos = 3;
        assert (utilities.nextPos(10) == expectedPos);
        assert (utilities.nextPos(20) == expectedPos);
        assert (utilities.nextPos(1) == expectedPos);
        assert (utilities.nextPos(30) == expectedPos);
        assert (utilities.nextPos(0) == expectedPos);
        utilities.nextPos(5);
        assert (utilities.nextPos(10) == 4);
        utilities.nextPos(20);
        assert (utilities.nextPos(30) == 5);
        utilities.nextPos(1);
        assert (utilities.nextPos(2) == 6);
        // Additional tests as per requirement
    }
}
