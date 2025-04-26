package com.densebrain.rif.client;

import java.lang.reflect.Field;
import java.rmi.RemoteException;
import java.util.Hashtable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class RIFManagerFactory_getManager_1_0_Test {

    private RIFManagerFactory rifManagerFactory;

    @BeforeEach
    void setUp() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        rifManagerFactory = RIFManagerFactory.getInstance();
        // Clear the managerMap to ensure a clean state for each test
        Field managerMapField = RIFManagerFactory.class.getDeclaredField("managerMap");
        managerMapField.setAccessible(true);
        Hashtable<String, RIFManager> managerMap = (Hashtable<String, RIFManager>) managerMapField.get(rifManagerFactory);
        managerMap.clear();
    }

    @Test
    void testGetManager_NullUrl() throws RemoteException {
        assertThrows(NullPointerException.class, () -> rifManagerFactory.getManager(null));
    }
}
