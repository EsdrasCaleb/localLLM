package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_getActor_3_1_Test {

    @Test
    public void testGetActor() {
        Starring starring = new Starring();
        String[] actors = { "actor1", "actor2", "actor3" };
        starring.setActor(actors);
        assertEquals("actor1", starring.getActor(0));
        assertEquals("actor2", starring.getActor(1));
        assertEquals("actor3", starring.getActor(2));
        assertEquals(null, starring.getActor(3));
    }
}
