package net.kencochrane.a4j.beans;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import java.util.ArrayList;
import org.junit.Before;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Starring_toString_4_0_Test {

    Starring star;

    @Before
    public void init() {
        star = new Starring();
    }

    @Test
    public void testToString() {
        String result = star.toString();
        assertEquals("Actors is null or size 0", result);
    }
}
