package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_toString_4_0_Test {

    @Test
    public void testToString() {
        Starring starring = new Starring();
        starring.setActor(new String[] { "Actor1", "Actor2" });
        String expectedOutput = "# of Actors = 2\n" + "Actor - Actor1\n" + "Actor - Actor2\n";
        assertEquals(expectedOutput, starring.toString());
    }
}
