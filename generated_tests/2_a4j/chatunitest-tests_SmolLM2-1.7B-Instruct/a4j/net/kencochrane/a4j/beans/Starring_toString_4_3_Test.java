package net.kencochrane.a4j.beans;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@RunWith(MockitoJUnitRunner.class)
public class Starring_toString_4_3_Test {

    @Mock
    private Starring focal;

    @InjectMocks
    private Starring testStarring;

    @Test
    public void testToString_ValidActorList() {
        testStarring.setActor(new String[] { "Actor 1", "Actor 2" });
        assertEquals("# of Actors = 2\nActor - Actor 1\nActor - Actor 2", testStarring.toString());
    }

    @Test
    public void testToString_NullActorList() {
        assertEquals("Actors is null or size 0", testStarring.toString());
    }
}
