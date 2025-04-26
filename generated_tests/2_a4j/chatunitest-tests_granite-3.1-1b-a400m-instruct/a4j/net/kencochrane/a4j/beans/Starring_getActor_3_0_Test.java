package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_getActor_3_0_Test {

    @Test
    void testGetActor() {
        Starring starring = new Starring();
        starring.setActor(new String[] { "Tom Cruise", "Meryl Streep" });
        assertEquals("Tom Cruise", starring.getActor(0));
        assertEquals("Meryl Streep", starring.getActor(1));
        assertEquals("null", starring.getActor(2));
    }
}
