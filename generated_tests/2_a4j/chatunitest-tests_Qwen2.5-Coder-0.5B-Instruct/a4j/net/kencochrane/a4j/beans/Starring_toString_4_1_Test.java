package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Starring_toString_4_1_Test {

    @Test
    public void testToString() {
        Starring starring = new Starring();
        starring.setActor(new String[] { "John Doe", "Jane Smith" });
        assertEquals("Actors is null or size 0\nActor - John Doe\nActor - Jane Smith", starring.toString());
    }
}
