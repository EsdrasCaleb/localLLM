package com.hf.sfm.system.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.business.PersonInfoMgr;
import com.hf.sfm.system.pdo.AWorker;
import com.hf.sfm.util.DaoFactory;

public class WorkerMgr_deleteByIds_1_0_Test {

    @Test
    public void deleteByIds() {
        WorkerMgr workerMgr = mock(WorkerMgr.class);
        String[] idnos = { "1001", "1002" };
        // Call the method under test
        String result = workerMgr.deleteByIds(idnos);
        // Assert the expected outcome
        assertEquals("1", result);
    }
}
