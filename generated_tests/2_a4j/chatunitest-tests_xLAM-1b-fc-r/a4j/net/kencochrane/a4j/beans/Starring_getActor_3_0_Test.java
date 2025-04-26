package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_getActor_3_0_Test {

    Starring starring = new Starring();

    @Test
    public void testGetActor() {
        starring.setActor(new String[] { "actor1", "actor2", "actor3" });
        assertEquals("actor1", starring.getActor(0));
        assertEquals("actor2", starring.getActor(1));
        assertEquals("actor3", starring.getActor(2));
    }
}
