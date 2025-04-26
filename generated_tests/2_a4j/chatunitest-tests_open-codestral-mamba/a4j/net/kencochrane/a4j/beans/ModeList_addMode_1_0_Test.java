package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ModeList_addMode_1_0_Test {

    @Test
    public void testAddMode() {
        ModeList modeList = new ModeList();
        Mode mode = Mockito.mock(Mode.class);
        modeList.addMode(mode);
        ArrayList<Mode> modes = (ArrayList<Mode>) modeList.getAllModes();
        assertEquals(1, modes.size());
        assertEquals(mode, modes.get(0));
    }
}
